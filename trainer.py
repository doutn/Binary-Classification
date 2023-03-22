import torch
import time

from model import count_parameters


class Trainer():
    def __init__(self, train_loader, test_loader, model, loss_fn, optimizer, scheduler, device, output_dir):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir


    def train_loop(self):
        size = len(self.train_loader.dataset)
        lowest_loss = 100
        self.model.to(self.device)
        for batch, (X, y) in enumerate(self.train_loader):
            # Compute prediction and loss
            X = X.to(self.device)
            y = y.to(self.device)

            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                if lowest_loss > loss:
                    lowest_loss = loss
        
        return lowest_loss


    def test_loop(self):
        size = len(self.test_loader.dataset)
        num_batches = len(self.test_loader)
        test_loss, correct = 0, 0
        self.model.to(self.device)

        with torch.no_grad():
            best_acc = 0
            for X, y in self.test_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size

        if best_acc < 100*correct:
            best_acc = 100*correct
        
        return best_acc
        

    def train(self, epochs):
        before_loss = 100
        before_acc = 0

        sum_start = time.time()
        for t in range(epochs):
            start = time.time()
            lowest_loss = self.train_loop()
            best_acc = self.test_loop()

            if before_loss > lowest_loss:
                before_loss = lowest_loss
                torch.save(self.model.state_dict(), self.output_dir + 'model_weights.pth')
            
            if before_acc < best_acc:
                before_acc = best_acc
            
            end = time.time()
            minute = int(end-start) // 60
            sec = int(end-start) % 60
            print(f"{t + 1}/{epochs}의 학습시간: {minute}분, {sec}초 입니다.")

        sum_end = time.time()
        minute = int(sum_end-sum_start) // 60
        sec = int(sum_end-sum_start) % 60
        num_parameters = count_parameters(self.model)
        print(f"총 {epochs}의 학습시간: {minute}분 {sec}초이며, lowest_loss: {before_loss:.4f}, best_acc: {before_acc:.2f}입니다.")
        print(f"모델의 총 파라미터 개수: {num_parameters}")
